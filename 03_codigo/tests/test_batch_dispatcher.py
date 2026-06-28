import threading
import time

import pytest

from video_search.batch_dispatcher import BatchDispatcher


def test_flush_on_batch_size():
    calls = []

    def process_fn(batch):
        calls.append(list(batch))
        return [x * 2 for x in batch]

    # Itens enfileirados ANTES de start() para o primeiro _collect_batch()
    # ja encontrar o lote completo -- deterministico, sem depender de timing.
    dispatcher = BatchDispatcher(process_fn=process_fn, batch_size=3, timeout_ms=5000)
    futures = [dispatcher.submit(i) for i in range(3)]
    dispatcher.start()

    results = [f.result(timeout=5) for f in futures]

    assert results == [0, 2, 4]
    assert len(calls) == 1
    assert calls[0] == [0, 1, 2]
    dispatcher.stop()


def test_flush_on_timeout_with_partial_batch():
    calls = []

    def process_fn(batch):
        calls.append(list(batch))
        return list(batch)

    dispatcher = BatchDispatcher(process_fn=process_fn, batch_size=10, timeout_ms=50)
    dispatcher.start()

    future = dispatcher.submit("only-one")
    result = future.result(timeout=5)

    assert result == "only-one"
    assert calls == [["only-one"]]
    dispatcher.stop()


def test_batch_mixes_items_from_multiple_sources():
    calls = []

    def process_fn(batch):
        calls.append(list(batch))
        return list(batch)

    dispatcher = BatchDispatcher(process_fn=process_fn, batch_size=4, timeout_ms=5000)
    # Simula 2 "cameras" submetendo antes do dispatcher começar a drenar.
    futures = [dispatcher.submit(("camA", i)) for i in range(2)]
    futures += [dispatcher.submit(("camB", i)) for i in range(2)]
    dispatcher.start()

    for f in futures:
        f.result(timeout=5)

    assert len(calls) == 1
    sources = {payload[0] for payload in calls[0]}
    assert sources == {"camA", "camB"}
    dispatcher.stop()


def test_process_fn_exception_propagates_to_all_futures_in_batch():
    def process_fn(batch):
        raise RuntimeError("falha no lote")

    dispatcher = BatchDispatcher(process_fn=process_fn, batch_size=2, timeout_ms=5000)
    futures = [dispatcher.submit(i) for i in range(2)]
    dispatcher.start()

    for f in futures:
        with pytest.raises(RuntimeError, match="falha no lote"):
            f.result(timeout=5)
    dispatcher.stop()


def test_stop_drains_pending_items():
    calls = []

    def process_fn(batch):
        calls.append(list(batch))
        return list(batch)

    dispatcher = BatchDispatcher(process_fn=process_fn, batch_size=10, timeout_ms=50)
    dispatcher.start()
    futures = [dispatcher.submit(i) for i in range(3)]

    dispatcher.stop()  # deve drenar os 3 itens antes de retornar

    assert all(f.done() for f in futures)
    assert sum(len(batch) for batch in calls) == 3
