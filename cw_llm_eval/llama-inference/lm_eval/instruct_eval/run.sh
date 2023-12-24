export OPENAI_API_KEY="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"
export ORGANIZATION="org-ykL6XJ3eydRkR7ghoE92aJAK"
export OPENAI_API_BASE="https://shanghai-chat-service-open-api.ai2open.win/v1"

rm -rf outputs
mkdir -p outputs/honest
mkdir -p outputs/helpful
mkdir -p outputs/harmness

python eval.py -data instruct_data/honest/honest.json -config configs/bloomz-nemo-sft-honest.yaml -output outputs/honest
sleep 300
python eval.py -data instruct_data/helpfulness/helpfulness.json -config configs/bloomz-nemo-sft-helpful.yaml -output outputs/helpful
sleep 300
python eval.py -data instruct_data/harmness/harmness.json -config configs/bloomz-nemo-sft-harm.yaml -output outputs/harmness
