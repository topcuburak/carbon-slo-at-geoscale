#!/usr/bin/env bash
#
# Shared topology configuration for local (non-K8s) runs.
# Source this from any start-* script to get endpoint maps.
#
# Usage:  source scripts/env-local.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- Topology ----
NUM_SCHEDULERS=2
NUM_DCS_PER_SCHED=1
NUM_DCS=$((NUM_SCHEDULERS * NUM_DCS_PER_SCHED))

# ---- Port ranges ----
PROV_PORT="${PROV_PORT:-50000}"
DC_BASE_PORT="${DC_BASE_PORT:-51000}"
SCHED_BASE_PORT="${SCHED_BASE_PORT:-52000}"

# ---- Build endpoint JSON maps ----

# All DCs: {"dc-0": "localhost:51000", "dc-1": "localhost:51001", ...}
DC_ENDPOINTS="{"
for i in $(seq 0 $((NUM_DCS - 1))); do
    [[ $i -gt 0 ]] && DC_ENDPOINTS+=","
    DC_ENDPOINTS+="\"dc-${i}\": \"localhost:$((DC_BASE_PORT + i))\""
done
DC_ENDPOINTS+="}"
echo "DC endpoints: $DC_ENDPOINTS"

# All schedulers: {"sched-0": "localhost:52000", ...}
SCHED_ENDPOINTS="{"
for i in $(seq 0 $((NUM_SCHEDULERS - 1))); do
    [[ $i -gt 0 ]] && SCHED_ENDPOINTS+=","
    SCHED_ENDPOINTS+="\"sched-${i}\": \"localhost:$((SCHED_BASE_PORT + i))\""
done
SCHED_ENDPOINTS+="}"
echo "Scheduler endpoints: $SCHED_ENDPOINTS"

# Provisioner needs both DCs and schedulers in one map
PROV_ENDPOINTS="$(echo "$DC_ENDPOINTS" | sed 's/}$//' ),$(echo "$SCHED_ENDPOINTS" | sed 's/^{//')"
echo "Provisioner endpoints: $PROV_ENDPOINTS"

echo "PROJECT_DIR: $PROJECT_DIR"
echo "NUM_SCHEDULERS: $NUM_SCHEDULERS"
echo "NUM_DCS_PER_SCHED: $NUM_DCS_PER_SCHED"
echo "NUM_DCS: $NUM_DCS"
echo "DC_BASE_PORT: $DC_BASE_PORT"
echo "SCHED_BASE_PORT: $SCHED_BASE_PORT"
echo "PROV_PORT: $PROV_PORT"
echo "DC_ENDPOINTS: $DC_ENDPOINTS"
echo "SCHED_ENDPOINTS: $SCHED_ENDPOINTS"
echo "PROV_ENDPOINTS: $PROV_ENDPOINTS"

export PROJECT_DIR NUM_SCHEDULERS NUM_DCS_PER_SCHED NUM_DCS
export DC_BASE_PORT SCHED_BASE_PORT PROV_PORT
export DC_ENDPOINTS SCHED_ENDPOINTS PROV_ENDPOINTS
