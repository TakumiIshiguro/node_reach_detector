for i in `seq 1`
do
  roslaunch node_reach_detector sim.launch world_name:=tsudanuma2-3.world initial_pose_x:=-5.0 initial_pose_y:=7.7 initial_pose_a:=3.14 robot_x:=-5.0 robot_y:=7.7 robot_Y:=3.14 gui:=true
  sleep 10
done