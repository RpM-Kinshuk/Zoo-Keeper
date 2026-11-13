"""Bounded range acquisition for this fixed task panel, using existing HubReader."""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock


class AcquisitionStopped(BaseException):
    """A strict limit that network retry handlers must not consume."""


class ResponseBudget:
    def __init__(self, used=0, limit=30_000_000_000):
        if not 0 <= used <= limit:
            raise ValueError('invalid inherited byte budget')
        self.reserved, self.limit = used, limit
        self.lock = Lock()

    def reserve(self, count):
        with self.lock:
            if count < 0 or self.reserved+count > self.limit:
                raise AcquisitionStopped('original response-byte budget exhausted')
            self.reserved += count

    def transport(self, method, url, headers):
        from evidence.hub import _transport
        if method != 'HEAD':
            start, end = map(int,headers['Range'][6:].split('-'))
            self.reserve(end-start+2)
        return _transport(method,url,headers)


def append_ranges(path, size, fetch, *, chunk=32*1024**2, workers=4, progress=None):
    """Append a contiguous prefix using bounded batches, never sparse holes."""
    start = path.stat().st_size if path.exists() else 0
    if not 0 <= start <= size or chunk < 1 or workers not in range(1,5):
        raise ValueError('invalid range plan')
    stats = {}
    with path.open('ab') as stream, ThreadPoolExecutor(max_workers=workers) as pool:
        while start < size:
            ranges = [(i,min(i+chunk,size)) for i in range(start,min(start+workers*chunk,size),chunk)]
            futures = [pool.submit(fetch,a,b) for a,b in ranges]
            for (a,b),future in zip(ranges,futures):
                data,read_stats = future.result()
                if len(data) != b-a:
                    raise ValueError('short resumed range')
                stream.write(data)
                stream.flush()
                start = b
                for key,value in read_stats.items():
                    stats[key] = stats.get(key,0)+value
                if progress:
                    progress(len(data),read_stats)
    return stats
