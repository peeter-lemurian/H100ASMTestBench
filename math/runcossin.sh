./cossin_test --dump-inputs ./trigunarytest.in
../torch/torchunary.py --op sin --file ./trigunarytest.in
../torch/torchunary.py --op cos --file ./trigunarytest.in
./cossin_test --op sin --torchinductor torchinductorsin.bin --torcheager torcheagersin.bin --verbose --quiet --color | less -R
./cossin_test --op cos --torchinductor torchinductorcos.bin --torcheager torcheagercos.bin --verbose --quiet --color | less -R
