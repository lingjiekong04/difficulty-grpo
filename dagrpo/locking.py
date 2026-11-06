"""OS advisory locks are released even when the platform kills the process."""
from contextlib import contextmanager
from pathlib import Path

@contextmanager
def exclusive_lock(path, blocking=False):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open('a+b')
    try:
        if stream.tell()==0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            if __import__('os').name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except (OSError,BlockingIOError) as exc:
            raise RuntimeError(f'Another job holds lock: {path}') from exc
        yield
    finally:
        # Closing the file descriptor releases the lock. Keep the inode for waiters.
        stream.close()
