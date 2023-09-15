for n in $(ls MaxText/configs/*s31_2*); do echo;echo; diff -u --color MaxText/configs/int8-s31-a1-q_TTF-clT.yml $n; done
