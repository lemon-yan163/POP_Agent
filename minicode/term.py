"""stdin 的唯一读取者。

问题：审批提示和 REPL 都想要一行输入，而这些请求可能被 Ctrl-C 取消。用
asyncio.to_thread(input, ...) 的做法是「每次要输入就开一个线程去读」，线程阻塞在
read(2) 里没法取消；一旦取消，它仍然挂在 stdin 上，于是在 REPL 打出的下一行会被这个
僵尸线程抢走（用户看到的就是「我打的字没了」）。退出时 asyncio.run 还要等这个线程，
进程甚至卡住。

做法：只用一个常驻 daemon 线程调用 input()。谁要输入就往队列里投一个请求，线程读到行后
把行交给「行到达的瞬间仍在等待的那个请求」：

- 同一时刻只有一个读者，队列保证一行只归一个请求，不会再有抢输入。
- 如果原本等这一行的请求已经被 Ctrl-C 取消，就把行转交给队列里下一个还在等的请求
  （通常是刚刚回到的 REPL 提示符），而不是丢掉——否则「中断一次就必然吞掉下一行输入」，
  用户得重打。交接判定在 pump 线程里做（它是唯一的写者），避免行还在路上时被别的请求插队，
  造成「先打的行反而后被处理」。
- 一个还在等的请求都没有时，这一行是在一个已经消失的提示符上打的，无处安放：丢弃并提示，
  不猜它的归属。
- daemon 线程让解释器能正常退出，不必等一个永远阻塞在 read(2) 的线程。
"""

import asyncio
import queue
import sys
import threading


class _Req:
    __slots__ = ("prompt", "future")

    def __init__(self, prompt: str, future: asyncio.Future) -> None:
        self.prompt = prompt
        self.future = future


_EOF = object()


class StdinBroker:
    def __init__(self) -> None:
        self._requests: "queue.Queue[_Req | None]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._gate: asyncio.Lock | None = None
        self._gate_loop: asyncio.AbstractEventLoop | None = None

    # ---- 供 event loop 线程调用 -------------------------------------------

    async def read_line(self, prompt: str = "") -> str:
        """读一行；等价于 input(prompt)，但可以被安全取消。"""
        loop = asyncio.get_running_loop()
        if self._gate is None or self._gate_loop is not loop:
            self._gate = asyncio.Lock()   # 绑定到第一个用它的 loop，避免模块级强引用
            self._gate_loop = loop
        if self._gate.locked():
            # 同时有两处要输入必然是编码错误：拿到的行会归属不明，宁可炸掉
            raise RuntimeError("stdin already being read by another reader")
        async with self._gate:
            if self._closed:
                raise EOFError
            self._ensure_pump(loop)
            fut: asyncio.Future = loop.create_future()
            self._requests.put(_Req(prompt, fut))
            return await fut

    # ---- 内部 -------------------------------------------------------------

    def _ensure_pump(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._thread is not None:
            return
        t = threading.Thread(target=self._pump, args=(loop,), name="stdin-pump", daemon=True)
        self._thread = t
        t.start()

    def _pump(self, loop: asyncio.AbstractEventLoop) -> None:
        while True:
            req = self._requests.get()
            if req is None:
                return
            if req.future.done():
                continue  # 请求方已经放弃，连 prompt 都不打，免得留下没人读的提示符
            try:
                # prompt 交给 input() 而不是自己 print：GNU readline 要靠它算光标列宽，
                # 否则方向键重绘会把提示符擦掉。
                line: object = input(req.prompt)
            except EOFError:
                self._closed = True
                self._deliver(loop, req.future, _EOF)
                continue
            except BaseException as e:
                # UnicodeDecodeError 之类：这一行坏了，但不能让 pump 线程死掉，
                # 否则之后所有输入都永久卡住。把异常交给读方处理（__main__ 会丢弃该行）。
                self._deliver(loop, req.future, e)
                continue
            self._deliver(loop, req.future, line)

    def _deliver(self, loop: asyncio.AbstractEventLoop, fut: asyncio.Future, value: object) -> None:
        """把读到的东西交给「此刻仍在等」的请求：原主已取消就往后顺延。"""
        target = self._landing(fut)
        if target is None:
            if value is not _EOF and not isinstance(value, BaseException):
                self._post(loop, self._note_dropped, value)
            return
        self._post(loop, self._resolve, target, value)

    def _landing(self, fut: asyncio.Future) -> asyncio.Future | None:
        while fut.done():
            try:
                nxt = self._requests.get_nowait()
            except queue.Empty:
                return None
            if nxt is None:
                return None
            # 顺延到的请求同样可能已经取消，继续往后找
            fut = nxt.future
        return fut

    def _post(self, loop: asyncio.AbstractEventLoop, fn, *args) -> None:
        try:
            loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:
            pass  # loop 已经关了（正在退出），丢弃即可

    def _resolve(self, fut: asyncio.Future, value: object) -> None:
        if fut.done():
            # 极小窗口：判定时还活着，投递过程中被取消
            if value is not _EOF and not isinstance(value, BaseException):
                self._note_dropped(value)
            return
        if value is _EOF:
            fut.set_exception(EOFError())
        elif isinstance(value, BaseException):
            fut.set_exception(value)
        else:
            fut.set_result(value)

    def _note_dropped(self, value: object) -> None:
        preview = value if len(value) <= 40 else value[:40] + "…"
        print(
            f"\n[已丢弃一行输入: {preview!r} —— 打这一行时对应的提示符已取消，且没有别的地方在等输入]",
            file=sys.stderr,
            flush=True,
        )


stdin = StdinBroker()
