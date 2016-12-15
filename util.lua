require 'image'

function displayTensors(tensors, win)
   if not display then
      return
   end
   local todisp, n = {}, nil
   for i = 1, #tensors do
      for j = 1, tensors[i]:size(1) do
	 todisp[1+#todisp] = tensors[i][j]
      end
      n = n or tensors[i]:size(1)
   end
   local x = image.toDisplayTensor{input = todisp, padding = 2, nrow = n}
   local win_out = display.image(x, {legend='', win=win})
   return win_out
end