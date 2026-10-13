"""Run the real local explanation model against the bundled cancellation example."""
import argparse
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from llm_agent_eval.explanations import LocalExplanationClient, explain_result
from llm_agent_eval.service import IndexService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with TemporaryDirectory(prefix="repolens-explanation-smoke-") as tmp:
        service = IndexService(Path(tmp))
        try:
            row = service.submit("demo")
            while service.get(row["id"])["status"] in {"queued", "indexing"}:
                time.sleep(.05)
            result = service.search(row["id"], "Cancelling an order does not restore inventory.",
                                    "bm25", 4000, 3)
            client = LocalExplanationClient(args.device)
            complete = client.complete
            def record(prompt):
                raw = complete(prompt)
                Path("artifacts/explanation-raw.txt").write_text(raw, encoding="utf-8")
                return raw
            client.complete = record
            explained = explain_result(result, client)
            Path("artifacts").mkdir(exist_ok=True)
            Path("artifacts/explanation-smoke.json").write_text(
                json.dumps(explained, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(explained, indent=2))
        finally:
            service.close()


if __name__ == "__main__":
    main()
