#!/bin/bash
for filename in TestData/*.jpg; do
   ./a.out "$filename" 
done
