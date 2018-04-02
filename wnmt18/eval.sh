#!/usr/bin/env bash

# Change to match model name in train.sh:
MODEL="model.baseline"
# Change for contrastive runs with the same model:
RUN="default"

# CLI args for cpu, gpu, or both
if [[ ${#} < 1 ]]; then
  echo "Usage: ${0} [cpu] [gpu]"
  exit 2
fi

for DEVICE in ${@}; do
  # Choose optimal settings for CPU or GPU
  if [[ ${DEVICE} == "cpu" ]]; then
    DOCKER_RUN="docker run --init --rm -i -u $(id -u):$(id -g) -v $(pwd):/work -w /work sockeye:latest-cpu"
    DEVICE_ARGS="--use-cpu"
    BATCH_SIZE=1
    CHUNK_SIZE=1
    RESTRICT_ARGS="--restrict-lexicon=${MODEL}/top_k_lexicon"
  elif [[ ${DEVICE} == "gpu" ]];
    DOCKER_RUN="nvidia-docker run --init --rm -i -u $(id -u):$(id -g) -v $(pwd):/work -w /work sockeye:latest-gpu"
    DEVICE_ARGS=""
    BATCH_SIZE=32
    CHUNK_SIZE=1000
    RESTRICT_ARGS=""
  fi
  # Decode both test sets
  for SET in newstest2014 newstest2015; do
    # BPE encode -> decode -> BPE join
    time cat data/${SET}.en \
      |${DOCKER_RUN} apply_bpe.py -c codes \
      |${DOCKER_RUN} python3 -m sockeye.translate \
        --models=${MODEL} \
        --beam-size=5 \
        --batch-size=${BATCH_SIZE} \
        --chunk-size=${CHUNK_SIZE} \
        --length-penalty-alpha=0.1 \
        --length-penalty-beta=0.0 \
        --max-output-length-num-stds=2 \
        --bucket-width=10 \
        ${DEVICE_ARGS} \
        ${RESTRICT_ARGS} \
      |sed -u -r 's/@@( |$)//g' \
      >${MODEL}/${SET}.${RUN}.${DEVICE} \
      2>${MODEL}/${SET}.${RUN}.${DEVICE}.time
    # Evaluate
    ${DOCKER_RUN} python3 -m sockeye.evaluate \
      --hypotheses ${MODEL}/${SET}.out.${DEVICE} \
      --references data/${SET}.de \
      --metrics bleu \
      >${MODEL}/${SET}.${RUN}.${DEVICE}.bleu
  done
done
