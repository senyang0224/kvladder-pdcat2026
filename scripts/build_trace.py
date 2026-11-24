"""Build the conference burst inputs from the preprocessed SCBench QA parquet."""
import argparse
import hashlib
import json
from pathlib import Path


PREFIX_DIGEST = {
    32768: "e9a5a78325aab29e4f520a072882a5143b9c9ee1f80cc2ae7df29777791afa9a",
    65536: "9b1439171e827dba654cf92648a823c7af26b23a5ae97c93f9c1c2a62f46c223",
}
DATA_SHA256 = "6408de16ad0b6d1ffab9fd3f89180f44996f19fb2734bf9b840bb3249032c010"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="local Llama-3.1-8B-Instruct directory")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("traces"))
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    if hashlib.sha256(args.parquet.read_bytes()).hexdigest() != DATA_SHA256:
        parser.error("the QA parquet differs from the conference input (see REPRODUCTION.md)")
    model = str(args.model.resolve())
    encode = AutoTokenizer.from_pretrained(model, local_files_only=True).encode
    session = pq.read_table(args.parquet, columns=["prompts", "ground_truth"]).slice(0, 1).to_pylist()[0]
    document = encode(session["prompts"][0], add_special_tokens=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for length, expected in PREFIX_DIGEST.items():
        prefix = document[:length]
        digest = hashlib.sha256(b"".join(t.to_bytes(8, "little", signed=True) for t in prefix)).hexdigest()
        if digest != expected:
            parser.error(f"tokenizer produced a different {length}-token prefix")
        requests = []
        for index in range(8):
            turn = index % 5
            query = session["prompts"][turn + 1]
            text = f"\n\nQuestion: {query}\nAnswer:"
            tokens = encode(text, add_special_tokens=False)
            requests.append(dict(request_id=f"conf-{index:03d}", arrival_sequence=index, arrival_offset_s=0.,
                                 turn_index=turn, query=query, query_text=text, query_token_ids=tokens,
                                 prompt_tokens=length + len(tokens), gold_answers=[str(session["ground_truth"][turn])]))
        trace = dict(schema="kvladder-conf-branch-trace-v1", model=model, tokenizer=model,
                     source=dict(workload="SCBench/scbench_qa_eng", session_index=0, dataset_sha256=DATA_SHA256),
                     context_tokens=length, token_prefix_digest=digest, prefix_token_ids=prefix, requests=requests,
                     generation=dict(temperature=0., max_new_tokens=32, stop=[]), deadline_s=600.,
                     arrival_process=dict(kind="deterministic_burst", interarrival_ms=0.))
        payload = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
        trace["content_sha256"] = hashlib.sha256(payload).hexdigest()
        path = args.output_dir / f"qa-{length}.json"
        path.write_text(json.dumps(trace) + "\n")
        print(f"{path}  prefix SHA-256 {digest}")


if __name__ == "__main__":
    main()
