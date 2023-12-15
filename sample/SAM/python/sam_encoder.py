import sophon.sail as sail
import logging
logging.basicConfig(level=logging.INFO)


class SamEncoder():
    def __init__(self,args,img_size = 1024) -> None:
        # load bmodel
        self.net = sail.Engine(args.embedding_bmodel, args.dev_id, sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_shape = self.net.get_input_shape(self.graph_name,self.input_name)
        self.output_name = self.net.get_output_names(self.graph_name)[0]
        self.output_shape = self.net.get_output_shape(self.graph_name,self.output_name) 

        self.img_size = img_size
        logging.debug("load {} success!".format(args.bmodel))
        logging.debug(str(("graph_name: {}, input_names & input_shapes: ".format(self.graph_name), self.input_name, self.input_shape)))
        logging.debug(str(("graph_name: {}, output_names & output_shapes: ".format(self.graph_name), self.output_name, self.output_shape)))
    
    def embedding(self, input):
        input_data = {self.input_name:input}
        outputs = self.net.process(self.graph_name, input_data)
        return outputs[self.output_name] # (1, 256, 64, 64)
