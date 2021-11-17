#!/usr/bin/gawk -f

BEGIN {
    print "Atom x y vx vy fx fy omegax omegaz tqz diameter mass"
}
{
    if ($1 == "1" && NF == 12) {
        print $0;
    }
}
