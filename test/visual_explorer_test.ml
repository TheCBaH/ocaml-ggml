open Model_explorer
open Jsont_bytesrw

let%expect_test "keyValue" =
  let json = Result.get_ok @@ encode_string KeyValue.jsont @@ KeyValue.create ~key:"boo" ~value:"bar" in
  print_string json;
  [%expect {| {"key":"boo","value":"bar"} |}];
  ()
