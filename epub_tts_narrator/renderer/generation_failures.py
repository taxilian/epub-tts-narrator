def drama3_truncation(report):
    if report.get("passed") is not False:
        return False
    if report.get("reasons") == ["generation_incomplete"]:
        return True
    return (report.get("error_type") in {"EngineError", "FishHTTPError"}
            and "did not finish the text" in str(report.get("error", "")))
