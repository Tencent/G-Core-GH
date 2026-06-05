# 关闭透明大页，并清空 page cache（tlinux 特殊姿势）。
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag
sync
echo 3 >/proc/sys/vm/drop_caches
echo 1 >/proc/sys/vm/compact_memory

sysctl -w vm.memory_qos=1
sysctl -w vm.pagecache_limit_global=1 
echo 30 > /proc/sys/vm/pagecache_limit_ratio
echo 1 > /proc/sys/vm/pagecache_limit_async
