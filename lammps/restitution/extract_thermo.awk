#!/usr/bin/gawk -f

function is_number( str ) {
    return str + 0 == str;
}

BEGIN {
  header_nf=-1;
  header_lineno=0;
}
{
  if ($1 == "Step") {
     header_nf = NF;
     header_lineno = NR;
     if (ENVIRON["NEW_FILE"] == "") {
         print $0;
     }
  }
  if (NR > header_lineno && NF == header_nf) {
      num_n = 0
      for (i=1; i<=NF; i++) {
          if (is_number($i)) {
              num_n ++;
          }
      }

      if (num_n == NF)
          print $0;
  }
}
