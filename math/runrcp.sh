./rcp_test --dump-inputs ./rcptest.in
../torch/torchunary.py --op reciprocal --file ./rcptest.in
./rcp_test --torchinductor torchinductorreciprocal.bin --torcheager torcheagerreciprocal.bin --verbose --quiet --color | less -R
