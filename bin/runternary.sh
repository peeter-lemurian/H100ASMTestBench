complexops/ternary_test --dump-inputs ./ternarytest.in
for i in mac clip sel ; do
  ../bin/torchternary.py --op $i --file ./ternarytest.in
done
#exit
for i in mac clip sel ; do
  complexops/ternary_test --$i --torchinductor torchinductor$i.bin --torcheager torcheager$i.bin --quiet --csv > $i.csv
  complexops/ternary_test --$i --torchinductor torchinductor$i.bin --torcheager torcheager$i.bin --quiet --color | tee $i.out
  #complexops/ternary_test --$i --torchinductor torchinductor$i.bin --torcheager torcheager$i.bin --quiet --color | less -R
done
