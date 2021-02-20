#!/bin/bash

num_vertex="3"
p="0.9"
eps="10"
tightening_rate="0"
method="0"

for i in {1..100}; do

    ./app/build_graph $i $num_vertex test_planar_$i
    sleep .1
    ./app/search_graph test_planar_$i $p $eps $tightening_rate $method test_search_$i
done