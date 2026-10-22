set -eu
iptables-legacy -N ARENA_SC71_FORWARD 2>/dev/null || true
iptables-legacy -N ARENA_SC71_INPUT 2>/dev/null || true
iptables-legacy -F ARENA_SC71_FORWARD
iptables-legacy -F ARENA_SC71_INPUT
iptables-legacy -A ARENA_SC71_FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables-legacy -A ARENA_SC71_FORWARD -s 172.31.72.0/24 -d 172.31.72.10 -p tcp --dport 8081 -j ACCEPT
iptables-legacy -A ARENA_SC71_FORWARD -s 172.31.72.0/24 -j REJECT
for target in 10:8080 11:8080 12:8080 20:5000 30:8081 40:3128; do
  address=${target%:*}
  port=${target#*:}
  iptables-legacy -A ARENA_SC71_FORWARD -s 172.31.71.0/24 -d "172.31.71.$address" -p tcp --dport "$port" -j ACCEPT
done
iptables-legacy -A ARENA_SC71_FORWARD -s 172.31.71.0/24 -j REJECT
iptables-legacy -A ARENA_SC71_FORWARD -j RETURN
iptables-legacy -A ARENA_SC71_INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables-legacy -A ARENA_SC71_INPUT -s 172.31.71.0/24 -j REJECT
iptables-legacy -A ARENA_SC71_INPUT -s 172.31.72.0/24 -j REJECT
iptables-legacy -A ARENA_SC71_INPUT -j RETURN
iptables-legacy -C DOCKER-USER -j ARENA_SC71_FORWARD 2>/dev/null || iptables-legacy -I DOCKER-USER 1 -j ARENA_SC71_FORWARD
iptables-legacy -C INPUT -j ARENA_SC71_INPUT 2>/dev/null || iptables-legacy -I INPUT 1 -j ARENA_SC71_INPUT
