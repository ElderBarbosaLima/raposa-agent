from raposa.grounding import check_scores, extract_scores


def test_extracts_common_score_formats() -> None:
    text = "Ganhamos de 2 x 1, depois 3-0 e empatamos 1×1."
    assert extract_scores(text) == {(2, 1), (3, 0), (1, 1)}


def test_ignores_dates_and_times() -> None:
    assert extract_scores("Jogo em 2099-05-24 às 17:00, rodada 8") == set()


def test_grounded_answer_passes_in_either_order() -> None:
    tool_output = '{"score": "0-1", "goals_for": 1, "goals_against": 0}'
    report = check_scores("Vencemos o São Paulo por 1 x 0 fora de casa!", [tool_output])
    assert report.ok
    assert report.checked == 1


def test_invented_score_is_flagged() -> None:
    report = check_scores("Foi 5-0 pra Raposa!", ['{"score": "2-0"}'])
    assert not report.ok
    assert report.ungrounded == [(5, 0)]
