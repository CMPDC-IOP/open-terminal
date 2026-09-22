"""File pages stay bounded and can be joined without losing text."""

import asyncio
import io

import httpx
import pytest

from open_terminal.utils.fs import MAX_READ_BYTES, MAX_READ_LINES, UserFS, read_text_page


def page(text, **kwargs):
    return read_text_page(io.StringIO(text, newline=None), **kwargs)


@pytest.mark.parametrize("text", ["", "one", "one\n", "one\r\ntwo\rthree", "\n\n"])
def test_small_files_and_newlines(text):
    result = page(text)
    expected = text.replace("\r\n", "\n").replace("\r", "\n")
    assert result["content"] == expected
    assert result["total_lines"] == len(expected.splitlines())
    assert not result["truncated"]
    assert result["next_start_line"] is None


def test_explicit_range_cannot_bypass_line_limit():
    result = page("a\nb\nc\nd", start_line=2, end_line=100, max_lines=2)
    assert result["content"] == "b\nc\n"
    assert result["truncation_reason"] == "lines"
    assert (result["next_start_line"], result["next_start_column"]) == (4, 1)
    assert result["total_lines"] == 4


def test_explicit_short_range_reports_remaining_file():
    result = page("a\nb\nc", end_line=1)
    assert result["content"] == "a\n"
    assert result["truncated"]
    assert result["next_start_line"] == 2


@pytest.mark.parametrize("separator_offset", [8191, 8192, 8193])
def test_special_line_break_at_chunk_boundary(separator_offset):
    text = "a" * separator_offset + "\u2028second\vthird"
    result = page(text, start_line=2, max_lines=1)
    assert result["content"] == "second\v"
    assert result["total_lines"] == 3
    assert (result["next_start_line"], result["next_start_column"]) == (3, 1)


@pytest.mark.parametrize(
    "text", [
        "汉字🙂" * 10 + "\nend", "abc\ndef\n", "abcd", "abcd\n",
        "ab\u2028汉\v🙂\x85last\f", "\v\f\x1c\x1d\x1e\x85\u2028\u2029",
    ]
)
def test_byte_limited_pages_round_trip_with_unicode_and_long_lines(text):
    params = {}
    parts = []
    for _ in range(len(text) + 1):
        result = page(text, max_bytes=4, max_lines=2, **params)
        assert len(result["content"].encode("utf-8")) <= 4
        assert result["total_lines"] == len(text.splitlines())
        parts.append(result["content"])
        if not result["truncated"]:
            break
        cursor = (result["next_start_line"], result["next_start_column"])
        assert cursor > (params.get("start_line", 1), params.get("start_column", 1))
        params = dict(start_line=cursor[0], start_column=cursor[1])
    else:
        pytest.fail("Pagination did not finish")
    assert "".join(parts) == text


def test_long_lines_are_read_in_bounded_chunks():
    class BoundedStream(io.StringIO):
        def readline(self, size=-1):
            assert 0 < size <= 8192
            return super().readline(size)

        def read(self, *args):
            pytest.fail("Must not read the whole stream")

    text = "🙂" * 30000 + "\nlast"
    result = read_text_page(BoundedStream(text), max_bytes=16)
    assert result["content"] == "🙂" * 4
    assert result["next_start_column"] == 5
    assert result["total_lines"] == 2


def test_resume_beyond_internal_chunk_boundary():
    text = "🙂" * 30000
    result = page(text, start_column=20000, max_bytes=12)
    assert result["content"] == "🙂" * 3
    assert result["next_start_column"] == 20003


@pytest.fixture
def route(tmp_path, monkeypatch):
    from open_terminal import env

    monkeypatch.setattr(env, "API_KEY", "fixture-key")
    from open_terminal import main

    fs = UserFS(home=str(tmp_path))
    main.app.dependency_overrides[main.get_filesystem] = lambda: fs
    main.app.dependency_overrides[main.verify_api_key] = lambda: None

    def get(**params):
        async def request():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app), base_url="http://test"
            ) as client:
                return await client.get("/files/read", params=params)

        return asyncio.run(request())

    yield get
    main.app.dependency_overrides.pop(main.get_filesystem)
    main.app.dependency_overrides.pop(main.verify_api_key)


def test_path_only_request_is_bounded_and_resumable(route, tmp_path):
    get = route
    content = ("x" * 57 + "\n") * 24360
    (tmp_path / "references.bib").write_text(content)
    result = get(path="references.bib").json()
    assert result["total_lines"] == 24360
    assert result["truncated"]
    assert len(result["content"].encode()) <= MAX_READ_BYTES
    assert result["end_line"] <= MAX_READ_LINES
    second = get(
        path="references.bib",
        start_line=result["next_start_line"],
        start_column=result["next_start_column"],
    ).json()
    assert (result["content"] + second["content"]) == content[
        : len(result["content"]) + len(second["content"])
    ]


def test_document_extraction_uses_same_limits(route, tmp_path, monkeypatch):
    from open_terminal.utils import documents

    get = route
    (tmp_path / "document.pdf").write_bytes(b"\xffPDF")
    text = "entry\v" * (MAX_READ_LINES + 1)
    monkeypatch.setattr(
        documents, "EXTRACTORS", [("application/pdf", ".pdf", lambda _: text)]
    )
    result = get(path="document.pdf").json()
    assert result["content"] == "entry\v" * MAX_READ_LINES
    assert result["total_lines"] == MAX_READ_LINES + 1
    assert result["truncated"]
    assert result["next_start_line"] == MAX_READ_LINES + 1


def test_binary_image_still_returns_binary(route, tmp_path):
    get = route
    data = b"\x89PNG\r\n\x1a\n"
    (tmp_path / "image.png").write_bytes(data)
    response = get(path="image.png")
    assert response.headers["content-type"] == "image/png"
    assert response.content == data


def test_route_validates_ranges_and_missing_files(route, tmp_path):
    get = route
    (tmp_path / "text.txt").write_text("text")
    assert get(path="text.txt", start_line=3, end_line=2).status_code == 400
    assert get(path="text.txt", start_column=20).status_code == 400
    assert get(path="text.txt", start_line=0).status_code == 422
    assert get(path="missing").status_code == 404

