complexops/unaryint32_test --dump-inputs unaryint32_test.in
#for i in popcount clz ctz ; do
#../bin/torchunary.py --op $i --file unaryint32_test.in
#done
#exit
for i in popcount clz ctz ; do
#complexops/unaryint32_test --$i --torchinductor torchinductor$i.bin --torcheager torcheager$i.bin --quiet --csv > $i.csv
#complexops/unaryint32_test --$i --torchinductor torchinductor$i.bin --torcheager torcheager$i.bin --quiet --color #| less -R
complexops/unaryint32_test --$i --quiet --color #| less -R
done
