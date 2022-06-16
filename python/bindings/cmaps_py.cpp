

template<typename COP, typename OBJ>
void def_inner(pybind11::module& m){
  m.def(("inner_"+COP::shortname()).c_str(),
    [](OBJ& R, const OBJ& x, const OBJ& y){
      COP op;
      cnine::InnerBiCmap(op,R,x,y);
    });
}

template<typename COP, typename OBJ>
void def_outer(pybind11::module& m){
  m.def(("outer_"+COP::shortname()).c_str(),
    [](OBJ& R, const OBJ& x, const OBJ& y){
      COP op;
      cnine::OuterBiCmap(op,R,x,y);
    });
}

template<typename COP, typename OBJ>
void def_cellwise(pybind11::module& m){
  m.def(("cellwise_"+COP::shortname()).c_str(),
    [](OBJ& R, const OBJ& x, const OBJ& y){
      COP op;
      cnine::CellwiseBiCmap(op,R,x,y);
    });
}


template<typename COP, typename OBJ>
void def_mprod(pybind11::module& m){
  m.def(("mprod_"+COP::shortname()).c_str(),
    [](OBJ& R, const OBJ& x, const OBJ& y){
      COP op;
      cnine::MprodBiCmap(op,R,x,y);
    });
}


template<typename COP, typename OBJ>
void def_convolve1(pybind11::module& m){
  m.def(("convolve1_"+COP::shortname()).c_str(),
    [](OBJ& R, const OBJ& x, const OBJ& y){
      COP op;
      cnine::Convolve1BiCmap(op,R,x,y);
    });
}


template<typename COP, typename OBJ>
void def_convolve2(pybind11::module& m){
  m.def(("convolve2_"+COP::shortname()).c_str(),
    [](OBJ& R, const OBJ& x, const OBJ& y){
      COP op;
      cnine::Convolve2BiCmap(op,R,x,y);
    });
}


