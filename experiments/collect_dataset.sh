for i in `seq 1`
do
  roslaunch node_reach_detector sim.launch world_name:=tsudanuma2-3.world  robot_x:=0.0 robot_y:=0.0 robot_Y:=0.0 gui:=true
  sleep 10
done