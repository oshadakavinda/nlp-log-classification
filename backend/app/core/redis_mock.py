import sqlite3
import os

class SQLiteRedisMock:
    def __init__(self, db_path=None):
        if db_path is None:
            backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_path = os.path.join(backend_dir, "celery_redis_mock.db")
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS lists (
                key TEXT,
                value TEXT,
                idx INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def set(self, key, value, *args, **kwargs):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, str(value)))
        conn.commit()
        conn.close()
        return True

    def get(self, key, *args, **kwargs):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT value FROM kv WHERE key = ?", (key,))
        row = c.fetchone()
        conn.close()
        if row:
            return row[0].encode('utf-8')
        return None

    def delete(self, key, *args, **kwargs):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM kv WHERE key = ?", (key,))
        c.execute("DELETE FROM lists WHERE key = ?", (key,))
        conn.commit()
        conn.close()
        return True

    def rpush(self, key, value, *args, **kwargs):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COALESCE(MAX(idx), -1) + 1 FROM lists WHERE key = ?", (key,))
        idx = c.fetchone()[0]
        c.execute("INSERT INTO lists (key, value, idx) VALUES (?, ?, ?)", (key, str(value), idx))
        conn.commit()
        conn.close()
        return True

    def lrange(self, key, start, end, *args, **kwargs):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT value FROM lists WHERE key = ? ORDER BY idx ASC", (key,))
        rows = c.fetchall()
        conn.close()
        return [row[0].encode('utf-8') for row in rows]

    def ltrim(self, key, start, end, *args, **kwargs):
        # Simply keep the last N items (e.g. 1000 items)
        if start == -1000 and end == -1:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT idx FROM lists WHERE key = ? ORDER BY idx DESC LIMIT 1 OFFSET 999", (key,))
            row = c.fetchone()
            if row:
                cutoff_idx = row[0]
                c.execute("DELETE FROM lists WHERE key = ? AND idx < ?", (key, cutoff_idx))
                conn.commit()
            conn.close()
        return True

    def ping(self):
        return True
