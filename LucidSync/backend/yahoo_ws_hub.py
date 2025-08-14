import asyncio, json, random, logging
from typing import Dict, Set, Optional
import yfinance as yf
from fastapi import WebSocket

log = logging.getLogger("yahoo-hub")

class YahooHub:
    def __init__(self):
        self.clients: Set[WebSocket] = set()
        self.symbols_by_client: Dict[WebSocket, Set[str]] = {}
        self._union: Set[str] = set()          # 현재 전체 구독 합집합 (내부 상태)
        self.subscribed: Set[str] = set()      # 실제 야후에 적용된 집합
        self.last_tick: Dict[str, dict] = {}   # 최근 틱 캐시

        self.ws: Optional[yf.AsyncWebSocket] = None
        self._listen_task: Optional[asyncio.Task] = None
        self._ready = asyncio.Event()          # 야후 WS 연결 준비 완료 신호
        self._lock = asyncio.Lock()
        self._closing = False

    async def start(self):
        if self._listen_task is None:
            self._listen_task = asyncio.create_task(self._listen_loop())

    async def _listen_loop(self):
        """
        끊기면 지수백오프로 자동 재연결. 매 재연결 때 현재 union을 다시 subscribe.
        """
        backoff = 1.0
        while not self._closing:
            try:
                log.info("Connecting to Yahoo streamer...")
                self.ws = yf.AsyncWebSocket()
                async with self.ws:
                    self._ready.set()
                    backoff = 1.0  # 성공하면 백오프 리셋
                    # 현재 합집합을 적용
                    await self._apply_union(force_all=True)
                    # 메시지 수신 루프
                    await self.ws.listen(self._on_message)
            except Exception as e:
                log.warning("Yahoo WS disconnected: %r", e)
            finally:
                self._ready.clear()
                self.ws = None
                # 재연결 대기 (최대 30초)
                await asyncio.sleep(backoff + random.random())
                backoff = min(backoff * 2, 30.0)

    async def shutdown(self):
        self._closing = True
        if self._listen_task:
            self._listen_task.cancel()
            with contextlib.suppress(Exception):
                await self._listen_task

    async def _on_message(self, msg: dict):
        sym = (msg.get("id") or msg.get("symbol") or msg.get("s") or "").upper()
        if sym:
            self.last_tick[sym] = msg
            # 해당 심볼을 구독 중인 클라에게만 전달
            for ws, syms in list(self.symbols_by_client.items()):
                if sym in syms:
                    await self._safe_send(ws, json.dumps(msg))
        else:
            # 심볼키가 없으면 전체 브로드캐스트(거의 없음)
            data = json.dumps(msg)
            for ws in list(self.clients):
                await self._safe_send(ws, data)

    async def _safe_send(self, ws: WebSocket, data: str):
        try:
            await ws.send_text(data)
        except Exception:
            await self.disconnect(ws)

    async def connect(self, ws: WebSocket, symbols: Set[str]):
        await ws.accept()
        await self.start()
        async with self._lock:
            self.clients.add(ws)
            self.symbols_by_client[ws] = set()
        await self.update_symbols(ws, symbols)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self.clients.discard(ws)
            self.symbols_by_client.pop(ws, None)
            # union 재계산
            self._union = set().union(*self.symbols_by_client.values()) if self.symbols_by_client else set()
        # 야후 구독 재적용 (unsubscribe 포함)
        await self._apply_union()

    async def update_symbols(self, ws: WebSocket, symbols: Set[str]):
        symbols = {s.upper() for s in symbols if s}
        async with self._lock:
            self.symbols_by_client[ws] = symbols
            # union 재계산
            self._union = set().union(*self.symbols_by_client.values()) if self.symbols_by_client else set()
        # 야후 구독 재적용
        await self._apply_union()
        # 새로 구독한 심볼의 최근 틱을 부트스트랩
        bootstrap = [self.last_tick[s] for s in symbols if s in self.last_tick]
        if bootstrap:
            await self._safe_send(ws, json.dumps({"type": "bootstrap", "data": bootstrap}))

    async def _apply_union(self, force_all: bool = False):
        """
        현재 union과 실제 구독(subscribed)의 차이를 계산해서 subscribe/unsubscribe.
        - 야후 WS 연결 준비가 될 때까지 대기
        - 연결이 끊기면 예외 무시(재연결 시 force_all로 다시 적용)
        - 메시지를 너무 자주 보내지 않도록 청크/스로틀
        """
        await self._ready.wait()  # 연결 준비될 때까지 대기

        to_sub = self._union if force_all else (self._union - self.subscribed)
        to_unsub = set() if force_all else (self.subscribed - self._union)

        async def _batched(items, n=200):
            items = list(items)
            for i in range(0, len(items), n):
                yield items[i:i+n]

        # 구독
        if to_sub:
            try:
                for batch in [b async for b in _batched(to_sub)]:
                    await self.ws.subscribe(batch)  # yfinance AsyncWebSocket API
                    await asyncio.sleep(0)          # 이벤트 루프 양보
                self.subscribed |= to_sub
            except Exception as e:
                # 연결 이슈면 다음 재연결에서 force_all로 다시 적용
                log.warning("subscribe failed: %r", e)

        # 구독 해제
        if to_unsub:
            try:
                for batch in [b async for b in _batched(to_unsub)]:
                    await self.ws.unsubscribe(batch)
                    await asyncio.sleep(0)
                self.subscribed -= to_unsub
            except Exception as e:
                log.warning("unsubscribe failed: %r", e)
