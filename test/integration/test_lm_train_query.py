# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import pytest
import logging

from test.common import run_train_translate, tmp_digits_dataset

logger = logging.getLogger(__name__)


_TRAIN_LINE_COUNT = 100
_DEV_LINE_COUNT = 10
_TEST_LINE_COUNT = 10
_TEST_LINE_COUNT_EMPTY = 2
_LINE_MAX_LENGTH = 15
_TEST_MAX_LENGTH = 20

ENCODER_DECODER_SETTINGS = [
     # LSTM language model
    ("--language-model"
     " --decoder rnn --num-layers 2 --rnn-cell-type lstm --rnn-num-hidden 21 --num-embed 21:13"
     " --batch-size 8 --loss cross-entropy --optimized-metric perplexity --max-updates 10"
     " --checkpoint-frequency 10 --optimizer adam --initial-learning-rate 0.01",
     "--language-model"),
     # Transformer language model
    ("--language-model"
     " --encoder rnn --decoder transformer --num-layers 2:2 --rnn-cell-type lstm --rnn-num-hidden 21 --num-embed 21:12"
     " --transformer-attention-heads 2 --transformer-model-size 12"
     " --transformer-feed-forward-num-hidden 32 --transformer-activation-type swish1"
     " --batch-size 8 --max-updates 10"
     " --checkpoint-frequency 10 --optimizer adam --initial-learning-rate 0.01",
     "--language-model")]


@pytest.mark.parametrize("train_params, lm_query_params", ENCODER_DECODER_SETTINGS)
def test_sorted_digits(train_params: str,
                       lm_query_params: str):
    """Task: model sequences of sorted digits"""

    with tmp_digits_dataset(prefix="test_sorted_seq_modeling",
                            train_line_count=_TRAIN_LINE_COUNT,
                            train_max_length=_LINE_MAX_LENGTH,
                            dev_line_count=_DEV_LINE_COUNT,
                            dev_max_length=_LINE_MAX_LENGTH,
                            test_line_count=_TEST_LINE_COUNT,
                            test_line_count_empty=_TEST_LINE_COUNT_EMPTY,
                            test_max_length=_TEST_MAX_LENGTH,
                            sort_target=True) as data:

        # Test model configuration, including the output equivalence of batch and no-batch decoding
        lm_query_params_batch = lm_query_params + " --batch-size 2"

        # Ignore return values (perplexity and BLEU) for integration test
        run_train_translate(train_params=train_params,
                            translate_params=lm_query_params,
                            translate_params_equiv=lm_query_params_batch,
                            train_source_path=data['target'],
                            train_target_path=data['target'],
                            dev_source_path=data['validation_target'],
                            dev_target_path=data['validation_target'],
                            test_source_path=data['test_target'],
                            test_target_path=data['test_target'],
                            max_seq_len=_LINE_MAX_LENGTH + 1,
                            work_dir=data['work_dir'])