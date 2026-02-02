from mysql.connector.pooling import MySQLConnectionPool

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "detection_db",
}

_DB_POOL = None


def init_db_pool():
    global _DB_POOL
    if _DB_POOL is None:
        _DB_POOL = MySQLConnectionPool(
            pool_name="fyp_pool",
            pool_size=5,
            **DB_CONFIG,
        )
    return _DB_POOL


def get_db_connection():
    pool = init_db_pool()
    return pool.get_connection()


def is_duplicate_entry_error(error):
    return getattr(error, "errno", None) == 1062
