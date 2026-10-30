set -eu
iptables-legacy -N ARENA_SC35_1_FORWARD 2>/dev/null || true
iptables-legacy -N ARENA_SC35_1_INPUT 2>/dev/null || true
iptables-legacy -F ARENA_SC35_1_FORWARD
iptables-legacy -F ARENA_SC35_1_INPUT
for source in 172.31.91.10 172.31.93.5 172.31.95.5 172.31.91.11 172.31.93.7 172.31.95.6 172.31.91.12 172.31.93.6 172.31.95.7; do
  for target in 172.31.91.20 172.31.93.20 172.31.95.2 172.31.95.3 172.28.0.2; do
    iptables-legacy -A ARENA_SC35_1_FORWARD -s "$source" -d "$target" -p tcp --dport 5000 -j REJECT
  done
done
iptables-legacy -A ARENA_SC35_1_FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables-legacy -A ARENA_SC35_1_FORWARD -s 172.31.92.0/24 -d 172.31.92.10 -p tcp --dport 8081 -j ACCEPT
iptables-legacy -A ARENA_SC35_1_FORWARD -s 172.31.92.0/24 -j REJECT
for target in 10:8080 11:8080 12:8080 20:5000 30:8081 40:3128; do
  address=${target%:*}
  port=${target#*:}
  iptables-legacy -A ARENA_SC35_1_FORWARD -s 172.31.91.0/24 -d "172.31.91.$address" -p tcp --dport "$port" -j ACCEPT
done
iptables-legacy -A ARENA_SC35_1_FORWARD -s 172.31.91.0/24 -j REJECT
iptables-legacy -A ARENA_SC35_1_FORWARD -j RETURN
iptables-legacy -A ARENA_SC35_1_INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables-legacy -A ARENA_SC35_1_INPUT -s 172.31.91.0/24 -j REJECT
iptables-legacy -A ARENA_SC35_1_INPUT -s 172.31.92.0/24 -j REJECT
iptables-legacy -A ARENA_SC35_1_INPUT -j RETURN
iptables-legacy -C DOCKER-USER -j ARENA_SC35_1_FORWARD 2>/dev/null || iptables-legacy -I DOCKER-USER 1 -j ARENA_SC35_1_FORWARD
iptables-legacy -C INPUT -j ARENA_SC35_1_INPUT 2>/dev/null || iptables-legacy -I INPUT 1 -j ARENA_SC35_1_INPUT
