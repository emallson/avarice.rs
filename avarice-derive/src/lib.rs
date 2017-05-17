extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;

use proc_macro::TokenStream;
use quote::Ident;

#[proc_macro_derive(Submodular)]
pub fn submodular(input: TokenStream) -> TokenStream {
    let s = input.to_string();
    let ast = syn::parse_derive_input(&s).unwrap();
    let expanded = expand_bounds(&ast,
                                 Ident::new("Submodular"),
                                 Ident::new("None"),
                                 Ident::new("Some(1.0)"));

    expanded.parse().unwrap()
}

#[proc_macro_derive(Supermodular)]
pub fn supermodular(input: TokenStream) -> TokenStream {
    let s = input.to_string();
    let ast = syn::parse_derive_input(&s).unwrap();
    let expanded = expand_bounds(&ast,
                                 Ident::new("Supermodular"),
                                 Ident::new("Some(1.0)"),
                                 Ident::new("None"));

    expanded.parse().unwrap()
}

#[proc_macro_derive(Modular)]
pub fn modular(input: TokenStream) -> TokenStream {
    let s = input.to_string();
    let ast = syn::parse_derive_input(&s).unwrap();
    let expanded = expand_bounds(&ast,
                                 Ident::new("Modular"),
                                 Ident::new("Some(1.0)"),
                                 Ident::new("Some(1.0)"));

    expanded.parse().unwrap()
}

fn expand_bounds(ast: &syn::DeriveInput, ty: Ident, low: Ident, high: Ident) -> quote::Tokens {
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    quote! {
        impl #impl_generics ::avarice::objective::curvature::#ty for #name #ty_generics #where_clause {}
        impl #impl_generics ::avarice::objective::curvature::Bounded for #name #ty_generics #where_clause {
            fn bounds() -> (Option<f64>, Option<f64>) {
                (#low, #high)
            }
        }
    }
}
