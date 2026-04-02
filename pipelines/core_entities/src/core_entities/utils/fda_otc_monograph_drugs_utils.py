from core_entities.utils.python_utils import ensure_string_list


def add_over_the_counter_status_if_needed(marketing_status: object, is_otc_monograph: bool) -> list[str]:
    statuses = ensure_string_list(marketing_status)
    if not is_otc_monograph:
        return statuses

    normalized_statuses = {status.lower().strip() for status in statuses}
    if "over-the-counter" not in normalized_statuses:
        statuses.append("Over-the-counter")
    return statuses
