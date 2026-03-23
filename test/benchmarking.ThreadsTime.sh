#!/bin/bash

# --- 配置区 ---
# THREADS=(1 2 4 8 12 16 20)
THREADS=(2 4 8 12 16 20)
IN_FILE="in.test-steinhardt"
METAD_PLUGIN_PATH=~/work/Metad-toy/build
# 确保 HILLS 文件在每轮开始前被清理，或者在每轮结束后重命名
HILLS_FILE="HILLS"

echo "开始 Metadynamics 性能基准测试..."
echo "测试目标线程数: ${THREADS[*]}"
echo "--------------------------------------"

for np in "${THREADS[@]}"
do
    echo "正在运行: mpirun -np $np (结果将保存为 log.lammps.${np}T1G)"
    
    # 生成随机种子
    RANDOM_SEED=$(date +%s%N | cut -b 11-18)
    
    # 1. 执行 LAMMPS
    # 使用 -log 指定输出文件名，或者运行完再重命名
    mpirun -np $np lmp \
        -var seed ${RANDOM_SEED} \
        -var METAD_PLUGIN_PATH ${METAD_PLUGIN_PATH} \
        -in ${IN_FILE}
    
    # 2. 运行分析脚本 (如果需要保留每轮的图，可以重命名 png)
    python plot_hills.py ${IN_FILE} ${HILLS_FILE}
    mv FES.png FES_${np}T1G.png 2>/dev/null
    
    # 3. 整理日志文件
    if [ -f log.lammps ]; then
        mv log.lammps log.lammps.${np}T1G
    else
        echo "警告: 未找到 log.lammps"
    fi

    # 4. 清理环境，为下一轮做准备
    # 删除重启文件、HILLS 文件以及可能的临时文件
    rm -f restart.latest restart.metad restart.normal restart.liquidus
    rm -f ${HILLS_FILE}
    
    echo "Rank $np 测试完成。"
    echo "--------------------------------------"
done

echo "所有测试已完成！请查看 log.lammps.*T1G 文件。"
grep "Loop time" log.lammps.*T1G