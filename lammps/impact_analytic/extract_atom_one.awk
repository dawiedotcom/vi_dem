#!/usr/bin/gawk -f

BEGIN {
    print "Atom x vx vy fx fy omegaz tqz diameter mass"
}
{
    if ($1 == "1" && NF == 10) {
        print $0;
    }
}
