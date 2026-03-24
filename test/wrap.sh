#!/bin/bash
if [ $OMPI_COMM_WORLD_RANK -eq 0 ]; then
    # 只有 Rank 0 被监控
    nsys profile    --force-overwrite true \
                    --trace=cuda,nvtx \
                    --delay=5 \
                    -o plugin_rank0 "$@"
else
    # 其他进程正常运行
    "$@"
fi