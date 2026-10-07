from types import SimpleNamespace

from llm_agent_eval.providers import Usage, _usage_from_response


def test_usage_parses_sdk_style_response():
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18)
    )
    assert _usage_from_response(response) == Usage(11, 7, 18)


def test_usage_handles_missing_metadata():
    assert _usage_from_response(SimpleNamespace(usage=None)) == Usage()
