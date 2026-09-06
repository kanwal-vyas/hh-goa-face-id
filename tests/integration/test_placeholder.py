"""
No integration tests exist yet in this milestone — there is nothing
external (blockchain RPC, search provider) to integrate with. This file
exists to keep tests/integration collectible by pytest and documents
what will land here:

    - Milestone 4 (BlockchainProvider local): register/retrieve
      round-trip against a local dev chain.
    - Milestone 6 (SearchProvider, authorized corpus): real query
      against the authorized demo corpus returns non-static results.
    - Milestone 8 (pipeline integration): full run against example data.
"""


def test_integration_suite_is_currently_empty_by_design():
    assert True
