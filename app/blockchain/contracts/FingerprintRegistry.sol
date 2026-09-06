// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title FingerprintRegistry
/// @notice Minimal integrity registry for HH Goa 2026 Task 3.
/// Stores a content fingerprint commitment (SHA-256 hash + metadata),
/// never the underlying content itself. One record per unique
/// content_hash; registering the same content_hash twice reverts,
/// giving deterministic duplicate-registration behavior.
contract FingerprintRegistry {
    struct Record {
        bytes32 contentHash;
        string algorithm;
        string canonicalizationVersion;
        bytes32 sourceReferenceHash;
        uint256 timestamp;
        bool exists;
    }

    mapping(bytes32 => Record) private records;

    event FingerprintRegistered(
        bytes32 indexed contentHash,
        string algorithm,
        string canonicalizationVersion,
        bytes32 sourceReferenceHash,
        uint256 timestamp
    );

    /// @notice Register a new content fingerprint commitment.
    /// @dev Reverts if this contentHash has already been registered —
    /// deterministic duplicate-registration behavior, not silently
    /// overwritten and not silently ignored.
    function register(
        bytes32 contentHash,
        string calldata algorithm,
        string calldata canonicalizationVersion,
        bytes32 sourceReferenceHash
    ) external returns (bool) {
        require(!records[contentHash].exists, "FingerprintRegistry: already registered");

        records[contentHash] = Record({
            contentHash: contentHash,
            algorithm: algorithm,
            canonicalizationVersion: canonicalizationVersion,
            sourceReferenceHash: sourceReferenceHash,
            timestamp: block.timestamp,
            exists: true
        });

        emit FingerprintRegistered(
            contentHash,
            algorithm,
            canonicalizationVersion,
            sourceReferenceHash,
            block.timestamp
        );

        return true;
    }

    /// @notice Retrieve a previously registered record.
    /// @dev Returns a Record with exists=false if nothing was registered
    /// for this contentHash — callers must check `exists`.
    function getRecord(bytes32 contentHash) external view returns (Record memory) {
        return records[contentHash];
    }
}
