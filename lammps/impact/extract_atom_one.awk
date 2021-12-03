#!/usr/bin/gawk -f

BEGIN {
    print "Atom x vx fx diameter mass"
}
{
    if ($1 == "1" && NF == 6) {
        print $0;
    }
}
