./math/rcp_test --dump-inputs ./rcptest.in
../bin/torchunary.py --op reciprocal --file ./rcptest.in
./math/rcp_test --torchinductor torchinductorreciprocal.bin --torcheager torcheagerreciprocal.bin --verbose --quiet --color | less -R
