"""
tests/integration/conftest.py

Shared fixtures for the integration test suite -- the tests in this
folder are the ones that need a REAL, disposable Postgres database,
unlike everything under tests/ (one level up), which runs fully offline.

ELI5: the offline tests one folder up check "does the recipe make sense
on paper." These integration tests check "does the oven actually cook
the food" -- they need a real oven (a real test database), not a fake
one, because they're testing SQL behavior (does an UPDATE really update
the same row instead of inserting a new one? does a guard column really
stop a value from being overwritten?) that no amount of pure-Python
testing can substitute for.

Why a separate folder instead of mixing these into tests/: pytest can
be pointed at a specific folder, which matters here since these tests
are slower (real DB round-trips) and need setup (a database + an
environment variable) a fresh clone of the repo won't have yet. Keeping
them physically separate makes "run the fast tests" (`pytest tests/
--ignore=tests/integration`) and "run everything" two easy, distinct
commands, instead of one pile of tests where some silently need extra
setup.

Setup required to actually run these tests:
    Set TEST_DATABASE_URL to a Postgres connection string pointing at a
    disposable database. A dedicated Neon branch is a good fit here --
    the project already uses Neon, and branches are free and fast to
    create/destroy (see docs/PLAN.md's stack table). NEVER point this
    at your real production database -- these tests insert and delete
    real rows.
"""

import os

import pytest
from sqlalchemy import bindparam, create_engine, text


@pytest.fixture(scope="session")
def db_engine():
    """
    A SQLAlchemy engine for the test database. Skips the whole
    integration suite (rather than erroring) if TEST_DATABASE_URL isn't
    set, so a plain `pytest` run elsewhere in the repo doesn't break on
    a machine that hasn't configured a test database yet.
    """
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set -- skipping integration tests")
    engine = create_engine(url)
    yield engine
    engine.dispose()


@pytest.fixture
def sample_fighters(db_engine):
    """
    Inserts three minimal, throwaway fighters -- a red corner, a blue
    corner, and a third fighter used to simulate a late replacement
    swap -- and deletes them (and anything referencing them) afterward.

    Deliberately bypasses fighter_resolution.py's matching logic here:
    these tests are about whether load_bout() writes correctly, not
    about fighter-name matching, which already has its own offline
    coverage in tests/test_fighter_resolution.py.

    Yields (fighter_red_id, fighter_blue_id, swap_in_fighter_id).
    """
    with db_engine.begin() as conn:
        red_id = conn.execute(
            text(
                "INSERT INTO fighters (real_name) VALUES ('Test Fighter Red') RETURNING id"
            )
        ).scalar_one()
        blue_id = conn.execute(
            text(
                "INSERT INTO fighters (real_name) VALUES ('Test Fighter Blue') RETURNING id"
            )
        ).scalar_one()
        swap_id = conn.execute(
            text(
                "INSERT INTO fighters (real_name) VALUES ('Test Fighter Swap-In') RETURNING id"
            )
        ).scalar_one()

    ids = (red_id, blue_id, swap_id)
    yield red_id, blue_id, swap_id

    with db_engine.begin() as conn:
        # Bouts must go first -- fighters/events are referenced by FK.
        conn.execute(
            text(
                "DELETE FROM bouts WHERE fighter_red_id IN :ids OR fighter_blue_id IN :ids"
            ).bindparams(bindparam("ids", expanding=True)),
            {"ids": ids},
        )
        conn.execute(
            text("DELETE FROM fighters WHERE id IN :ids").bindparams(
                bindparam("ids", expanding=True)
            ),
            {"ids": ids},
        )


@pytest.fixture
def sample_event(db_engine):
    """A minimal, throwaway event row, cleaned up (with its bouts) afterward."""
    with db_engine.begin() as conn:
        event_id = conn.execute(
            text("""
                INSERT INTO events (name, event_date, wikipedia_pageid)
                VALUES ('Test Event For Integration Tests', '2099-01-01', 999999999)
                RETURNING id
            """)
        ).scalar_one()

    yield event_id

    with db_engine.begin() as conn:
        conn.execute(text("DELETE FROM bouts WHERE event_id = :id"), {"id": event_id})
        conn.execute(text("DELETE FROM events WHERE id = :id"), {"id": event_id})
