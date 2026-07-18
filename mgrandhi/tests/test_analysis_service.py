from PIL import Image

from mgrandhi.backend import analysis_service
from mgrandhi.retrieval.pipeline import AnalysisResult, Detection


def _result() -> AnalysisResult:
    image = Image.new("RGB", (32, 24), "white")
    return AnalysisResult(
        annotated_image=image,
        detections=[
            Detection(
                crop_id=1,
                box=(1, 2, 12, 18),
                category="soft drinks",
                subcategory="cola",
                score=0.12,
                area=176,
            )
        ],
        num_items=1,
        distinct_categories=1,
        empty_pct=0.25,
        empty_label="Moderate",
        shelf_type="Category-specific",
        review_count=0,
        timings={"yolo_s": 0.1, "classify_s": 0.2, "boxes": 1},
    )


def test_givenSkuExtractionDisabled_whenAnalyzingShelf_thenReturnsSavedOperatorReport(
    monkeypatch,
    tmp_path,
):
    progress = []
    captured = {}

    def fake_analyze(image, **kwargs):
        kwargs["progress_callback"]("detected", 1, 1)
        kwargs["progress_callback"]("classified", 1, 1)
        captured.update(kwargs)
        return _result()

    monkeypatch.setenv("SKU_EXTRACT_DEFAULT", "0")
    monkeypatch.setenv("FEEDBACK_ASSET_DIR", str(tmp_path))
    monkeypatch.setattr(analysis_service.pipeline, "analyze_image", fake_analyze)
    monkeypatch.setattr(
        analysis_service.db,
        "save_scan",
        lambda result, filename, records: 42,
    )
    monkeypatch.setattr(
        analysis_service.db,
        "attach_scan_artifacts",
        lambda scan_id, source_path, crop_paths: captured.update(
            {
                "artifact_scan_id": scan_id,
                "source_path": source_path,
                "crop_paths": crop_paths,
            }
        ),
    )

    report = analysis_service.analyze_shelf(
        Image.new("RGB", (32, 24), "white"),
        "aisle.jpg",
        lambda stage, percent, message: progress.append((stage, percent, message)),
    )

    assert report["scan_id"] == 42
    assert report["summary"]["num_items"] == 1
    assert report["detections"][0]["sku_error"] == "sku_disabled"
    assert report["annotated_image"].startswith("data:image/jpeg;base64,")
    assert captured["max_crops"] == 60
    assert captured["artifact_scan_id"] == 42
    assert (tmp_path / "scan_42" / "source.jpg").is_file()
    assert (tmp_path / "scan_42" / "crop_0001.jpg").is_file()
    assert progress[-1][0] == "complete"


def test_givenSkuBackendFailure_whenAnalyzingShelf_thenKeepsCategoryResultsAndWarning(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("SKU_EXTRACT_DEFAULT", "1")
    monkeypatch.setenv("FEEDBACK_ASSET_DIR", str(tmp_path))
    monkeypatch.setattr(analysis_service.pipeline, "analyze_image", lambda *args, **kwargs: _result())
    monkeypatch.setattr(
        analysis_service,
        "get_sku_backend",
        lambda: (_ for _ in ()).throw(RuntimeError("VLM unavailable")),
    )
    monkeypatch.setattr(analysis_service.db, "save_scan", lambda *args, **kwargs: 9)
    monkeypatch.setattr(analysis_service.db, "attach_scan_artifacts", lambda *args, **kwargs: None)

    report = analysis_service.analyze_shelf(Image.new("RGB", (32, 24)), "shelf.png")

    assert report["scan_id"] == 9
    assert report["warning"] == "Some package details could not be read."
    assert report["detections"][0]["category"] == "soft drinks"
    assert report["detections"][0]["sku_needs_review"] == 1
