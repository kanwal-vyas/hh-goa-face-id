@"
import { defineConfig } from "hardhat/config";

export default defineConfig({
  solidity: "0.8.24",
  paths: {
    sources: "./app/blockchain/contracts",
  },
  networks: {
    hardhatMainnet: {
      type: "edr-simulated",
      chainType: "l1",
    },
  },
});
"@ | Set-Content -Encoding utf8 hardhat.config.js