def drama3_truncation(report):
    """Whether a failed report is Drama 3 not finishing its text.

    That is the engine's unfinished-text error, or a kept take the silence
    watch stopped (``generation_stopped``) that did not pass verification.
    """
    if report.get("passed") is not False:
        return False
    if report.get("generation_stopped") == "silence":
        return True
    if report.get("reasons") == ["generation_incomplete"]:
        return True
    return (report.get("error_type") in {"EngineError", "FishHTTPError"}
            and "did not finish the text" in str(report.get("error", "")))
