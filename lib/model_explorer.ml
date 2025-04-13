module KeyValue = struct
  type t = { key : string; value : string }

  let make key value = { key; value }
  let create ~key ~value = make key value
  let key t = t.key
  let value t = t.value

  let jsont =
    Jsont.Object.map ~kind:"KeyValue" make
    |> Jsont.Object.mem "key" Jsont.string ~enc:key
    |> Jsont.Object.mem "value" Jsont.string ~enc:value
    |> Jsont.Object.finish
end
