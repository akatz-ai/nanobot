import { listAgentChannelMappings, loadNanobotConfig } from "../utils/config.js";

export function listAgents(nanobotConfigPath: string): Record<string, unknown> {
  const config = loadNanobotConfig(nanobotConfigPath);
  const agents = listAgentChannelMappings(config).map((agent) => ({
    name: agent.resolvedName,
    display_name: agent.displayName,
    channel_id: agent.channelId,
  }));

  return {
    status: "ok",
    count: agents.length,
    agents,
  };
}
