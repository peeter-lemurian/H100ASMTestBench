./exp2f_test --dump-inputs ./exp2test.in
../torch/torchunary.py --op exp2 --file ./exp2test.in
./exp2f_test --torchinductor torchinductorexp2.bin --torcheager torcheagerexp2.bin --verbose --quiet --color | less -R
