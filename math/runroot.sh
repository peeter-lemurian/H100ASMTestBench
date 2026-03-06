./sqrtrsq_test --dump-inputs ./roottest.in
../torch/torchunary.py --op sqrt --file ./roottest.in
../torch/torchunary.py --op rsqrt --file ./roottest.in
./sqrtrsq_test --op sqrt --torchinductor torchinductorsqrt.bin --torcheager torcheagersqrt.bin --verbose --quiet --color | less -R
./sqrtrsq_test --op rsqrt --torchinductor torchinductorrsqrt.bin --torcheager torcheagerrsqrt.bin --verbose --quiet --color | less -R
