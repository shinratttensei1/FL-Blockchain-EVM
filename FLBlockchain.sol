// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";


contract SimpleFLBlockchain is Ownable, Pausable {
    
    // ══════════════════════════════════════════════════════════════════════
    // Simple block structure
    // ══════════════════════════════════════════════════════════════════════
    
    struct Block {
        uint256 blockNumber;
        uint256 flRound;
        string blockType;        // "LOCAL", "VOTE", "GLOBAL"
        bytes32 contentHash;     // Hash of data
        bytes32 previousHash;
        uint256 timestamp;
        address submitter;
    }
    
    Block[] public blocks;
    mapping(address => bool) public authorizedClients;
    
    event BlockAdded(
        uint256 indexed blockNumber,
        uint256 indexed flRound,
        string blockType,
        bytes32 contentHash
    );
    
    constructor() Ownable(msg.sender) {
        // Create genesis block
        blocks.push(Block({
            blockNumber: 0,
            flRound: 0,
            blockType: "GENESIS",
            contentHash: keccak256("FL Blockchain Genesis"),
            previousHash: bytes32(0),
            timestamp: block.timestamp,
            submitter: msg.sender
        }));
    }
    
    function authorizeClient(address client) external onlyOwner {
        authorizedClients[client] = true;
    }
    
    function revokeClient(address client) external onlyOwner {
        authorizedClients[client] = false;
    }
    
    function pause() external onlyOwner {
        _pause();  // Built-in from Pausable
    }
    
    function unpause() external onlyOwner {
        _unpause();  // Built-in from Pausable
    }
    
    function addBlock(
        uint256 flRound,
        string memory blockType,
        bytes memory data
    ) external whenNotPaused returns (uint256) {
        // Owner can add any block, clients can only add LOCAL blocks
        if (msg.sender != owner()) {
            require(authorizedClients[msg.sender], "Not authorized");
            require(
                keccak256(bytes(blockType)) == keccak256(bytes("LOCAL")),
                "Clients can only add LOCAL blocks"
            );
        }
        
        bytes32 contentHash = keccak256(data);
        bytes32 previousHash = blocks[blocks.length - 1].contentHash;
        
        blocks.push(Block({
            blockNumber: blocks.length,
            flRound: flRound,
            blockType: blockType,
            contentHash: contentHash,
            previousHash: previousHash,
            timestamp: block.timestamp,
            submitter: msg.sender
        }));
        
        uint256 newBlockNumber = blocks.length - 1;
        
        emit BlockAdded(newBlockNumber, flRound, blockType, contentHash);
        
        return newBlockNumber;
    }
    
    function getBlock(uint256 index) external view returns (Block memory) {
        require(index < blocks.length, "Block does not exist");
        return blocks[index];
    }
    
    function getBlockCount() external view returns (uint256) {
        return blocks.length;
    }
    
    function verifyChain() external view returns (bool) {
        for (uint256 i = 1; i < blocks.length; i++) {
            if (blocks[i].previousHash != blocks[i - 1].contentHash) {
                return false;
            }
        }
        return true;
    }
    
    function getLatestBlock() external view returns (Block memory) {
        return blocks[blocks.length - 1];
    }
}
