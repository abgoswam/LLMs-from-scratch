from dataloader import GPTDatasetV1, create_dataloader_v1

# --- 2.6 Data sampling with a sliding window ------------------------------


def test_dataset_sliding_window(gpt2_tokenizer):
    txt = "a b c d e f g"
    ids = gpt2_tokenizer.encode(txt)
    dataset = GPTDatasetV1(txt, gpt2_tokenizer, max_length=3, stride=2)

    assert len(dataset) == 2
    x0, y0 = dataset[0]
    x1, y1 = dataset[1]
    assert (x0.tolist(), y0.tolist()) == (ids[0:3], ids[1:4])
    assert (x1.tolist(), y1.tolist()) == (ids[2:5], ids[3:6])


def test_dataloader_stride_1(raw_text):
    data_iter = iter(create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False))
    x, y = next(data_iter)
    assert x.tolist() == [[40, 367, 2885, 1464]]
    assert y.tolist() == [[367, 2885, 1464, 1807]]
    x, y = next(data_iter)
    assert x.tolist() == [[367, 2885, 1464, 1807]]
    assert y.tolist() == [[2885, 1464, 1807, 3619]]


def test_dataloader_batch_8_stride_4(raw_text):
    inputs, targets = next(iter(create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)))
    assert inputs.shape == (8, 4)
    assert inputs[:2].tolist() == [[40, 367, 2885, 1464], [1807, 3619, 402, 271]]
    assert targets[:2].tolist() == [[367, 2885, 1464, 1807], [3619, 402, 271, 10899]]
